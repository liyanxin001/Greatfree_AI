package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * Created: 08/18/2024, Bing Li
 */
final class ExplodeGradients
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray M = manager.randomNormal(new Shape(4, 4));
			System.out.println("A single matrix: " + M);
			for (int i = 0; i < 100; i++)
			{
				M = M.dot(manager.randomNormal(new Shape(4, 4)));
			}

			System.out.println("after multiplying 100 matrices: " + M);
		}
	}
}

