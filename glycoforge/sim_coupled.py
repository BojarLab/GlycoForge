import numpy as np
from glycowork.motif.graph import subgraph_isomorphism
from glycoforge.utils import clr, invclr


def _build_coupling_directions(n_glycans, n_components, motif_rules, glycan_sequences,
                               motif_bias, seed):
  """Build a (n_glycans × n_components) coupling direction matrix with unit-norm columns.

  Each column is an independently drawn standard-normal vector, optionally re-weighted
  so that glycans matching any motif in *motif_rules* have larger entries before
  normalization. The direction values in *motif_rules* are ignored; only motif presence
  determines the weight boost. This concentrates the shared latent signal on biochemically
  relevant glycans while preserving the unit-norm property needed for the coupling
  injection to be scaled purely by *coupling_strength* × sigma.

  Parameters
  ----------
  n_glycans : int
  n_components : int
      Number of independent shared latent dimensions.
  motif_rules : dict or None
      {motif_string: any} — direction values ignored; keys used for subgraph_isomorphism.
  glycan_sequences : list of str or None
  motif_bias : float
      Added weight per matching motif; 0.0 = uniform, 1.0 = double weight per match.
  seed : int

  Returns
  -------
  U : np.ndarray, shape (n_glycans, n_components)
      Unit-norm columns.
  """
  rng = np.random.default_rng(seed)
  U = rng.standard_normal((n_glycans, n_components))
  if motif_rules is not None and glycan_sequences is not None:
    scores = np.ones(n_glycans)
    for idx, seq in enumerate(glycan_sequences[:n_glycans]):
      for motif in motif_rules:
        if subgraph_isomorphism(seq, motif):
          scores[idx] += motif_bias
    U *= scores[:, None]
  norms = np.linalg.norm(U, axis=0, keepdims=True)
  return U / np.where(norms > 1e-10, norms, 1.0)


def inject_coupling(Y_A_clr, Y_B_clr, coupling_strength, n_coupling_components,
                    coupling_motif_A, coupling_motif_B, coupling_motif_bias,
                    glycan_sequences_A, glycan_sequences_B, seed, verbose=False):
  """Inject shared latent variation into two CLR-space glycome matrices.

  A latent factor matrix Z ~ N(0, I) of shape (n_samples × n_components) is drawn
  once and added to both glycomes as:

      Y_A_clr  +=  coupling_strength * (Z @ U_A.T) * sigma_A
      Y_B_clr  +=  coupling_strength * (Z @ U_B.T) * sigma_B

  where U_A, U_B are unit-norm direction matrices built by
  _build_coupling_directions, and sigma_A/sigma_B are the per-glycan standard
  deviations of each CLR matrix before injection. Scaling by sigma ensures that the
  coupling magnitude is expressed in units of natural within-glycome variation, so
  coupling_strength=1.0 adds one standard deviation of shared signal per glycan
  per unit of Z. The induced cross-glycome HSIC is proportional to
  coupling_strength². The matrices are modified in-place and also returned.

  Parameters
  ----------
  Y_A_clr, Y_B_clr : np.ndarray, shape (n_samples, n_glycans_*)
      CLR-transformed clean data, modified in-place.
  coupling_strength : float
      Injection magnitude. 0 = no coupling; 1.0 ≈ one sigma of shared variation.
  n_coupling_components : int
  coupling_motif_A/B : dict or None
      Motif bias dictionaries for each glycome's direction matrix.
  coupling_motif_bias : float
  glycan_sequences_A/B : list of str or None
  seed : int
  verbose : bool

  Returns
  -------
  Y_A_clr, Y_B_clr : np.ndarray (modified in-place)
  Z : np.ndarray, shape (n_samples, n_coupling_components)
      The shared latent factor used, stored in metadata for reproducibility.
  """
  n_samples, n_glycans_A = Y_A_clr.shape
  n_glycans_B = Y_B_clr.shape[1]
  rng = np.random.default_rng(seed)
  Z = rng.standard_normal((n_samples, n_coupling_components))
  U_A = _build_coupling_directions(
    n_glycans_A, n_coupling_components, coupling_motif_A,
    glycan_sequences_A, coupling_motif_bias, seed + 1000
  )
  U_B = _build_coupling_directions(
    n_glycans_B, n_coupling_components, coupling_motif_B,
    glycan_sequences_B, coupling_motif_bias, seed + 2000
  )
  sigma_A = np.std(Y_A_clr, axis=0, ddof=1)
  sigma_B = np.std(Y_B_clr, axis=0, ddof=1)
  Y_A_clr += coupling_strength * (Z @ U_A.T) * sigma_A[None, :]
  Y_B_clr += coupling_strength * (Z @ U_B.T) * sigma_B[None, :]
  if verbose:
    print(f"  [Coupling] strength={coupling_strength}, components={n_coupling_components}")
    print(f"    U_A column norms (sanity check, should be 1): {np.linalg.norm(U_A, axis=0).round(4)}")
    print(f"    Z range: [{Z.min():.3f}, {Z.max():.3f}]")
  return Y_A_clr, Y_B_clr, Z
