package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/14/2024, Bing Li
 */
final class SoftmaxTester
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.randomNormal(new Shape(2, 5));
			NDArray Xprob = LNTraining.softmax(X);
			System.out.println(Xprob);
			System.out.println(Xprob.sum(new int[] { 1 })); // TODO Auto-generated method stub
		}
	}
}
