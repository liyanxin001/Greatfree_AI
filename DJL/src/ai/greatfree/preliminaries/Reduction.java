package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class Reduction
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4f);
			System.out.println(x);
			
			System.out.println("------------------------");
			System.out.println(x.sum());
			
			System.out.println("------------------------");
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);
			System.out.println(a.getShape());
			System.out.println(a.sum());
			
			System.out.println("------------------------");
			NDArray sumAxis0 = a.sum(new int[] {0});
			System.out.println(sumAxis0);
			System.out.println(sumAxis0.getShape());

			System.out.println("------------------------");
			NDArray sumAxis1 = a.sum(new int[] {1});
			System.out.println(sumAxis1);
			System.out.println(sumAxis1.getShape());

			System.out.println("------------------------");
			NDArray y = a.sum(new int[] {0, 1});
			System.out.println(y);

			System.out.println("------------------------");
			y = a.mean();
			System.out.println(y);
			y = a.sum().div(a.size());
			System.out.println(y);

			System.out.println("------------------------");
			y = a.mean(new int[] {0});
			System.out.println(y);
			y = a.sum(new int[] {0}).div(a.getShape().get(0));
			System.out.println(y);

			System.out.println("------------------------");
			y = a.sum(new int[] {1}, true);
			System.out.println(y);
			y = a.div(y);
			System.out.println(y);
			y = a.cumSum(0);
			System.out.println(y);
		}
	}
}
