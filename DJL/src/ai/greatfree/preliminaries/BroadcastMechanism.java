package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing Li
 */
final class BroadcastMechanism
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var a = manager.arange(3f).reshape(3, 1);
			var b = manager.arange(2f).reshape(1, 2);
			System.out.println(a);
			System.out.println("--------------------");
			System.out.println(b);
			
			System.out.println("--------------------");
			var z = a.add(b);
			System.out.println(z);
		}
	}
}
