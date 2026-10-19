package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/21/2024, Bing Li
 */
final class Pooling
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.arange(9f).reshape(3, 3);
			System.out.println(X);
			System.out.println(ConvolutionUtils.pool2d(manager, X, new Shape(2, 2), "max"));
			System.out.println(ConvolutionUtils.pool2d(manager, X, new Shape(2, 2), "avg"));
		}
	}
}
