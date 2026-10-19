package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/10/2024, Bing Li
 */
final class MatrixVectorProducts
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(5f).reshape(5, 1);
			System.out.println(a);
			
			System.out.println("------------------------");
			NDArray x = manager.arange(5f).reshape(1, 5);
			System.out.println(x);
			
			System.out.println(a.getShape());
			System.out.println(x.getShape());
			
			System.out.println("------------------------");
			NDArray y = a.dot(x);
			System.out.println(y);
		}
	}
}
