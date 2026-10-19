package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/08/2024, Bing Li
 */
final class DataManipulation
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var x = manager.arange(12);
			System.out.println(x);

			System.out.println("--------------------");

			Shape s = x.getShape();
			System.out.println(s);
			System.out.println(x.size());

			System.out.println("--------------------");

			x = x.reshape(3, 4);
			System.out.println(x);
			System.out.println("--------------------");

			x = x.reshape(-1, 4);
			System.out.println(x);

			System.out.println("--------------------");

			x = x.reshape(3, -1);
			System.out.println(x);

			System.out.println("--------------------");

			x = manager.create(new Shape(3, 4));
			System.out.println(x);

			System.out.println("--------------------");

			x = manager.zeros(new Shape(2, 3, 4));
			System.out.println(x);

			System.out.println("--------------------");
			
			x = manager.ones(new Shape(2, 3, 4));
			System.out.println(x);

			System.out.println("--------------------");

			x = manager.randomNormal(0f, 1f, new Shape(3, 4), DataType.FLOAT32);
			System.out.println(x);

			System.out.println("--------------------");

			x = manager.randomNormal(new Shape(3, 4));
			System.out.println(x);

			System.out.println("--------------------");

			x = manager.create(new float[] {2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));
			System.out.println(x);
		}
	}
}

