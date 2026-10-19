package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class Scalars
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.create(3f);
			System.out.println("x = " + x);
			NDArray y = manager.create(2f);
			System.out.println("y = " + y);

			NDArray z = x.add(y);
			System.out.println(z);
			
			z = x.mul(y);
			System.out.println(z);
			
			z = x.div(y);
			System.out.println(z);
			
			z = x.pow(y);
			System.out.println(z);
		}
	}
}
