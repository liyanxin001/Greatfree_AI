package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/18/2024, Bing Li
 */
final class DropoutTester
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray X = manager.arange(16f).reshape(2, 8);
			System.out.println(MLPUtils.dropoutLayer(manager, X, 0));
			System.out.println(MLPUtils.dropoutLayer(manager, X, 0.5f));
			System.out.println(MLPUtils.dropoutLayer(manager, X, 1.0f));			
		}
	}
}

