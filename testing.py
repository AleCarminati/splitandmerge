import unittest
import numpy as np

import SkeletonSplitAndMerge as sm

class TestMethods(unittest.TestCase):

	def test_NNIGHierarchy_compute_posterior_hypers(self):
		"""Test the normal functioning of the compute_posterior_hypers function
		of the NNIGHierarchy class.
		"""
		model = sm.NNIGHierarchy(1,5,3,2,1)
		data = np.array([81,-11,11,19,11,45,-19,73,98,44,-6,33,-25,16,-33,-100,
			-77,-78,-38,-25,40,86,69,-78,-33,74,7,42,22,-9,-29,-28,66,-19,-73,-87,
			-69,-25,-30,43,27,-29,-14,81,-86,-21,85,28,-43,75])
		self.assertAlmostEqual(model.compute_posterior_hypers(data)[0],\
			1.74545454545)
		self.assertEqual(model.compute_posterior_hypers(data)[1],5)
		self.assertEqual(model.compute_posterior_hypers(data)[2],28)
		self.assertAlmostEqual(model.compute_posterior_hypers(data)[3],\
			70682.2181818)

if __name__=="__main__":
	unittest.main()