import ast

d = open('gp_lite/gp.py').read()
i = d.rfind('def _nll_only')
head = d[:i].rstrip() + '\n'

nll = '''
    def _nll_only(self, log_theta):
        """Value of the negative log-marginal likelihood at ``log_theta``."""
        self._set_from_log(log_theta)
        X = self.X_train
        K = self.kernel.K(X, X) + self.noise * np.eye(len(X))
        L, ok = _safe_cholesky(K)
        if not ok:
            return 1e10
        alpha = cho_solve((L, True), self._y_train_n)
        nll = (0.5 * float(self._y_train_n @ alpha)
               + np.log(np.diag(L)).sum()
               + 0.5 * len(X) * np.log(2.0 * np.pi))
        return nll if np.isfinite(nll) else 1e10
'''

part2 = open('gp_lite/_gp_part2.py').read()
lines2 = part2.splitlines()
while lines2 and lines2[0].startswith('ll)'):
    lines2 = lines2[1:]
part2 = '\n'.join(lines2).strip() + '\n'

out = head + '\n' + nll + '\n' + part2
open('gp_lite/gp.py', 'w').write(out)
ast.parse(out)
print('OK', len(out))
