package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/20/2024, Bing Li
 */
final class ConvolutionCalculation
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.create(new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, new Shape(3, 3));
			NDArray K = manager.create(new float[] { 0, 1, 2, 3 }, new Shape(2, 2));
			System.out.println(ConvolutionUtils.corr2d(manager, X, K));
		}
	}
}

