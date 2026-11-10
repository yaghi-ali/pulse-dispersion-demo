import unittest
import numpy as np
from analysis import *

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=np.linspace(-40,40,4096,endpoint=False);self.dt=self.t[1]-self.t[0];self.a=np.exp(-self.t**2/2)
    def test_zero_length(self):np.testing.assert_allclose(propagate(self.a,self.dt,-20,0),self.a,atol=1e-14)
    def test_energy(self):
        b=propagate(self.a,self.dt,-20,.2);self.assertAlmostEqual(np.sum(abs(b)**2),np.sum(abs(self.a)**2),places=10)
    def test_analytical_width(self):
        for l in [.03,.1,.2]:self.assertAlmostEqual(rms_width(self.t,propagate(self.a,self.dt,-20,l)),np.sqrt(1+(20*l)**2)/np.sqrt(2),places=8)
    def test_invalid_dt(self):
        with self.assertRaises(ValueError):propagate(self.a,0,20,.1)
