package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/10/2024, Bing Li
 */
final class MatrixMulMatrix
{

	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);
			
			System.out.println("------------------------");
			NDArray b = manager.arange(12f).reshape(4, 3);
			System.out.println(b);
			
			System.out.println("------------------------");
			NDArray x = a.dot(b);
			System.out.println(x);
		}
	}
}
