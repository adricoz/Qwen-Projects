
## Formules à utiliser

Notations : n points, K = k(X, X) + σ_n² I, L = cholesky(K) (triangulaire inférieure), y_n les sorties normalisées.

- α = K⁻¹ y_n = Lᵀ \ (L \ y_n)   (`scipy.linalg.cho_solve((L, True), y_n)`).
- Log-vraisemblance marginale : log p(y_n | X) = −½ y_nᵀ α − Σᵢ log Lᵢᵢ − (n/2) log 2π.
  Remarque : y_nᵀ K⁻¹ y_n = ‖L⁻¹ y_n‖² (avec β = L \ y_n, c'est β·β).
- Prédiction en x* : k* = k(X, x*) ; moyenne μ* = k*ᵀ α ; v = L \ k* ; variance de f : k(x*, x*) − vᵀv.
- Leave-one-out (R&W éq. 5.12) : avec [K⁻¹]ᵢᵢ la diagonale de K⁻¹,
  μ₋ᵢ = y_n,i − αᵢ / [K⁻¹]ᵢᵢ et σ²₋ᵢ = 1 / [K⁻¹]ᵢᵢ (variance de y_i, bruit inclus).
- Retour aux unités de y : μ = μ_n · s + ȳ, écart-type × s, variance × s², avec ȳ et s = `np.std(y)` du jeu complet.
- `fit` minimise −log p(y_n | X) en fonction de θ = (log l₁..l_d, log σ², log σ_n²).
