package ai.greatfree.convolutionnn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/20/2024, Bing Li
 */
final class ImageObjectEdgeDetector
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.ones(new Shape(6, 8));
			X.set(new NDIndex(":" + "," + 2 + ":" + 6), 0f);
			System.out.println(X);
			
			System.out.println("------------------------");
			NDArray K = manager.create(new float[]{1, -1}, new Shape(1,2));
			System.out.println(K);
			
			System.out.println("------------------------");
			NDArray Y = ConvolutionUtils.corr2d(manager, X, K);
			System.out.println(Y);
			
			System.out.println("------------------------");
			NDArray Z = ConvolutionUtils.corr2d(manager, X.transpose(), K);
			System.out.println(Z);
		}
	}
}
