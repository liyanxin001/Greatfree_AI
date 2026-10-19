package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/09/2024, Bing Li
 */
final class DotProducts
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4f);
			System.out.println(x);
			
			System.out.println("------------------------");
			NDArray y = manager.ones(new Shape(4));
			System.out.println(y);

			System.out.println("------------------------");
			NDArray z = x.dot(y);
			System.out.println(z);

			System.out.println("------------------------");
			z = x.mul(y).sum();
			System.out.println(z);
		}
	}
}
