from pathlib import Path
import hashlib
import json
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class GeneratedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = pd.read_csv(ROOT / 'data/processed/telco_base_limpa.csv')
        cls.report = json.loads((ROOT / 'reports/analise.json').read_text(encoding='utf-8'))

    def test_counts_and_missing_values(self):
        self.assertEqual(self.base.shape, (7043, 20))
        self.assertEqual(self.base['Total Charges'].isna().sum(), 11)
        self.assertEqual(self.base.isna().sum().sum(), 11)
        self.assertEqual(self.base['Churn Value'].value_counts().to_dict(), {0: 5174, 1: 1869})
        forbidden = {'CustomerID','Churn Label','Churn Score','Churn Reason','CLTV'}
        self.assertFalse(forbidden.intersection(self.base.columns))

    def test_source_hashes(self):
        self.assertEqual(len(self.report['arquivos_origem']), 7)
        for name, sha in self.report['arquivos_origem'].items():
            self.assertEqual(hashlib.sha256((ROOT/'data/raw'/name).read_bytes()).hexdigest(), sha)

    def test_folds_and_train_only_statistics(self):
        validation = []
        assigned = pd.read_csv(ROOT/'data/processed/particoes.csv')
        numeric = self.base[['Tenure Months','Monthly Charges','Total Charges']].to_numpy()
        labels = self.base['Churn Value'].to_numpy()
        for k in range(1,6):
            with np.load(ROOT/f'data/processed/folds/fold_{k}.npz', allow_pickle=False) as f:
                tr, va = f['train_indices'], f['valid_indices']
                validation.extend(va.tolist())
                self.assertEqual(len(set(tr)&set(va)), 0)
                self.assertEqual(set(tr)|set(va), set(range(7043)))
                self.assertEqual(f['X_train'].shape, (len(tr),46))
                self.assertEqual(f['X_valid'].shape, (len(va),46))
                self.assertTrue(np.isfinite(f['X_train']).all())
                self.assertTrue(np.isfinite(f['X_valid']).all())
                np.testing.assert_array_equal(f['y_train'], labels[tr])
                np.testing.assert_array_equal(f['y_valid'], labels[va])
                np.testing.assert_array_equal(assigned.iloc[va]['validation_fold'], np.full(len(va),k))
                # Independently reconstruct statistics only from the training rows.
                median = np.nanmedian(numeric[tr],axis=0)
                train = np.where(np.isnan(numeric[tr]),median,numeric[tr])
                valid = np.where(np.isnan(numeric[va]),median,numeric[va])
                mean, std = train.mean(axis=0), train.std(axis=0,ddof=0)
                np.testing.assert_allclose(f['X_train'][:,:3],(train-mean)/std,atol=1e-10)
                np.testing.assert_allclose(f['X_valid'][:,:3],(valid-mean)/std,atol=1e-10)
                self.assertEqual(len(set(f['feature_names'])),46)
                self.assertTrue(set(np.unique(f['X_train'][:,3:])).issubset({0,1}))
                np.testing.assert_array_equal(f['X_train'][:,3:].sum(axis=1),np.full(len(tr),16))
        self.assertEqual(sorted(validation),list(range(7043)))


if __name__ == '__main__':
    unittest.main()
