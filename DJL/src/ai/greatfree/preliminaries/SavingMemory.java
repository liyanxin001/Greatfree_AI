package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDManager;

/**
 * Created: 08/09/2024, Bing li
 */
final class SavingMemory
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			var a = manager.arange(3f).reshape(3, 1);
			System.out.println("--------------------");
			a.add(2);
			System.out.println(a);

			a.addi(2);
			System.out.println("--------------------");
			System.out.println(a);
		}
	}
}
