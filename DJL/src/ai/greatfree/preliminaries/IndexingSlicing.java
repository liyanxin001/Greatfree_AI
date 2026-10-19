package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;

/**
 * Created: 08/09/2024, Bing Li
 */
final class IndexingSlicing
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var x = manager.arange(12);
			System.out.println(x);
			System.out.println("--------------------");
			
			var y = x.get(":-1");
			System.out.println(y);

			System.out.println("--------------------");
			y = x.get("1:3");
			System.out.println(y);
	
			System.out.println("--------------------");
			var z = manager.arange(20f).reshape(4, 5);
			System.out.println(z);

			System.out.println("--------------------");
			z.set(new NDIndex("1, 2"), 9);
			System.out.println(z);
			
			System.out.println("--------------------");
			z.set(new NDIndex("0:2, :"), 12);
			System.out.println(z);
		}
	}
}
