package ai.greatfree.testing;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 09/01/2024, Bing Li
 */
final class NDArrayMethods
{

	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray a = manager.arange(20f).reshape(5, 4);
			System.out.println(a);

			a.muli(2);
			
			System.out.println(a);
			
			NDArray b = manager.arange(20f).reshape(5, 4);
			byte[] bytesOfB = b.encode();
			
			
			
//			a.set(NDArray.decode(manager, bytesOfB));
			a.set(bytesOfB);
			
			System.out.println(a);
		}
	}
}

