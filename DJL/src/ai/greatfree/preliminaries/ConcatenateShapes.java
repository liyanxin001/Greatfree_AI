package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/09/2024, Bing Li
 */
final class ConcatenateShapes
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var x = manager.arange(12f).reshape(3, 4);
			var y = manager.create(new float[] {2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));
			System.out.println(x);
			System.out.println("--------------------");
			System.out.println(y);

			System.out.println("--------------------");
			var z = x.concat(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.concat(y, 1);
			System.out.println(z);
			
			System.out.println("--------------------");
			z = x.eq(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.sum();
			System.out.println(z);
		}
	}
}
