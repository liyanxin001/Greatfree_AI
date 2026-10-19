package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class Vectors
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4f);
			System.out.println(x);
			System.out.println(x.get(3));
			
			System.out.println(x.size(0));
			System.out.println(x.getShape());
		}
	}
}
