package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class DataOperations
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var x = manager.create(new float[] {1f, 2f, 4f, 8f});
			var y = manager.create(new float[] {2f, 2f, 2f, 2f});

			System.out.println(x);
			System.out.println("--------------------");
			System.out.println(y);

			System.out.println("--------------------");

			var z = x.add(y);
			System.out.println(z);
			
			System.out.println("--------------------");
			z = x.sub(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.mul(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.div(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.pow(y);
			System.out.println(z);

			System.out.println("--------------------");
			z = x.exp();
			System.out.println(z);
		}
	}
}
