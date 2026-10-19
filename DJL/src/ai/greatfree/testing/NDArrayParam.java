package ai.greatfree.testing;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/29/2024, Bing Li
 */
final class NDArrayParam
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(24f).reshape(2, 3, 4);
			System.out.println(x);
			x.setRequiresGradient(true);
//			x.setRequiresGradient(false);
			System.out.println(x.getGradient());
		}
	}
}

