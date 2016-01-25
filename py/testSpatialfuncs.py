import unittest
import numpy as np
import spatialfuncs as sf


class SimpleTests(unittest.TestCase):

	def testPiOne(self):
		"""test that get.pi returns appropriate values for cannonical test case 1 (equilateral triangle)"""

		x =np.array([[1.,0.,0.],[1.,1.,0.],[2.,0.5,np.sqrt(.75)]])

		#first no lower limit
		res2=sf.getPiTyped(x,1,2,1.5)
		self.assertEqual(res2, 0.5)

		res2=sf.getPiTyped(x,1,2,1.5,.5)
		self.assertEqual(res2, 0.5)


	def testPiTwo(self):
		"""test that get.pi returns appropriate values cannonical test case 2 (points on a line)"""

		x=np.array([[1,0,0], [2,1,0], [2,-1,0], [3,2,0], [2,-2,0], [3,3,0],[3,-3,0]])

		#pi 0,1.5 should be 1, 1.5-2.5 should be 0.5 and 2.5+ should be 0
		res2=sf.getPiTyped(x, 1, 2, [1.5,2.5,1000], [0,1.5,2.5])

		self.assertTrue(np.all(res2==[1,0.5,0]))


	def testThetaOne(self):
		"""test that get.theta returns appropriate values for cannonical test case 1 (equilateral triangle)"""

		x =np.array([[1.,0.,0.],[1.,1.,0.],[2.,0.5,np.sqrt(.75)]])

		#first no lower limit
		res2=sf.getThetaTyped(x,1,2,1.5)
		self.assertEqual(res2, 1.)

		res2=sf.getThetaTyped(x,1,2,1.5,.5)
		self.assertEqual(res2, 1.)


	def testThetaTwo(self):
		"""test that get.theta returns appropriate values cannonical test case 2 (points on a line)"""

		x=np.array([[1,0,0], [2,1,0], [2,-1,0], [3,2,0], [2,-2,0], [3,3,0],[3,-3,0]])

		res2=sf.getThetaTyped(x, 1, 2, [1.5,2.5,1000], [0,1.5,2.5])

		self.assertTrue(np.all(res2==[np.inf,1.,0.]))

if __name__ == '__main__':
    unittest.main()