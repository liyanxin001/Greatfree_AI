package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/11/2024, Bing Li
 */
final class Norms
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);
			
			System.out.println("------------------------");
			
			System.out.println(l2Norm(a));
			System.out.println(l1Norm(a));
		}
	}
	
	public static NDArray l2Norm(NDArray w)
	{
		return ((w.pow(2).sum()).sqrt());
	}
	
	public static NDArray l1Norm(NDArray w)
	{
		return w.abs().sum();
	}
}
