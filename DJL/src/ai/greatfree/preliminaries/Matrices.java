package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class Matrices
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);

			System.out.println("------------------------");
			NDArray b = a.transpose();
			System.out.println(b);
			
			System.out.println("------------------------");
			NDArray c = manager.create(new float[][] {{1, 2, 3}, {2, 0, 4}, {3, 4, 5}});
			System.out.println(c);
			System.out.println(c.eq(c.transpose()));
		}
	}
}

