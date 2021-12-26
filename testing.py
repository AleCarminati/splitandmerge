import unittest
import numpy as np

import SkeletonSplitAndMerge as sm

class TestMethods(unittest.TestCase):

	def test_NNIGHierarchy_compute_posterior_hypers_general(self):
		"""Test the normal functioning of the compute_posterior_hypers function
		of the NNIGHierarchy class.
		"""
		model = sm.NNIGHierarchy(1,5,3,2,1)
		data = np.array([81,-11,11,19,11,45,-19,73,98,44,-6,33,-25,16,-33,-100,
			-77,-78,-38,-25,40,86,69,-78,-33,74,7,42,22,-9,-29,-28,66,-19,-73,-87,
			-69,-25,-30,43,27,-29,-14,81,-86,-21,85,28,-43,75])
		self.assertAlmostEqual(model.compute_posterior_hypers(data)[0],\
			1.74545454545)
		self.assertEqual(model.compute_posterior_hypers(data)[1],55)
		self.assertEqual(model.compute_posterior_hypers(data)[2],28)
		self.assertAlmostEqual(model.compute_posterior_hypers(data)[3],\
			70682.2181818)

	def test_NNIGHierarchy_compute_posterior_hypers_one_data(self):
		""" Test the functioning of the compute_posterior_hypers function
		of the NNIGHierarchy class when it receives a single data point.
		"""
		model = sm.NNIGHierarchy(2,8,5,7,3)
		data = np.array([3422])
		self.assertEqual(model.compute_posterior_hypers(data)[0],382)
		self.assertEqual(model.compute_posterior_hypers(data)[1],9)
		self.assertEqual(model.compute_posterior_hypers(data)[2],5.5)
		self.assertEqual(model.compute_posterior_hypers(data)[3],5198407)

	def test_NNIGHierarchy_compute_posterior_hypers_no_data(self):
		""" Test that compute_posterior_hypers function of the NNIGHierarchy class
		launches an exception when it receives an empty dataset.
		"""
		model = sm.NNIGHierarchy(2,8,5,7,3)
		data= np.array([])
		with self.assertRaises(Exception):
			model.compute_posterior_hypers(data)

	def test_NNIGHierarchy_prior_pred_lpdf_general(self):
		"""Test the normal functioning of the prior_pred_lpdf function
		of the NNIGHierarchy class.
		"""
		model = sm.NNIGHierarchy(2,8,5,7,3)
		x = 2.5
		self.assertAlmostEqual(model.prior_pred_lpdf(x), np.log(0.2843239699768))

	def test_NNIGHierarchy_conditional_pred_lpdf_general(self):
		"""Test the normal functioning of the conditional_pred_lpdf function
		of the NNIGHierarchy class.
		"""
		model = sm.NNIGHierarchy(7,4,3,12,3)
		x = 14
		data = np.array([84,70,86,-100,-53,88,95,-39,95,61,87,74,-53,79,4,3,25,
			-45,86,82,23,-69,-22,86,-92,72,-53,-23,-13,27,-41,23,-97,14,-89,40,26,4,
			61,-28,25,90,30,43,-8,-82,-44,80,-37,87])
		self.assertAlmostEqual(model.conditional_pred_lpdf(x,data),\
			np.log(0.0068674610086127931507507))

	def test_NNIGHierarchy_conditional_pred_lpdf_no_data(self):
		""" Test that conditional_pred_lpdf function of the NNIGHierarchy class
		launches an exception when it receives an empty dataset.
		"""
		model = sm.NNIGHierarchy(7,4,3,12,3)
		x = 14
		data = np.array([])
		with self.assertRaises(Exception):
			model.conditional_pred_lpdf(x,data)


if __name__=="__main__":
	unittest.main()