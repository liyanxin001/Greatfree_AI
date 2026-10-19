package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class NDArrayArithmetic
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);
			
			System.out.println("------------------------");
			NDArray b = a.duplicate();
			System.out.println(b);
			
			System.out.println("------------------------");
			NDArray c = a.mul(b);
			System.out.println(c);
			
			System.out.println("------------------------");
			int x = 2;
			NDArray y = manager.arange(24f).reshape(2, 3, 4);
			System.out.println(y);

			System.out.println("------------------------");
			y = y.add(x);
			System.out.println(y);
			
			System.out.println("------------------------");
			System.out.println((y.mul(x)).getShape());
		}
	}
}
