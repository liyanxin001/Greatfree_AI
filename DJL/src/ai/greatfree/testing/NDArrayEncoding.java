package ai.greatfree.testing;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Created: 08/29/2024, Bing Li
 */
final class NDArrayEncoding
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(4);
			System.out.println(x);
			System.out.println("x length = " + x.encode().length);

			System.out.println("1) ----------------");
			NDArray y = manager.arange(8);
			System.out.println(y);
			System.out.println("y length = " + y.encode().length);
			
			System.out.println("2) ----------------");
			NDArray xb = NDArray.decode(manager, x.encode());
			System.out.println(xb);
			
			System.out.println("3) ----------------");
			NDArray yb = NDArray.decode(manager, y.encode());
			System.out.println(yb);
		}
		
		System.out.println("4) ----------------");
		NDManager m1 = NDManager.newBaseManager();
		NDArray x = m1.arange(4);
		System.out.println(x);
		System.out.println("x length = " + x.encode().length);
		byte[] xb = x.encode();
		m1.close();
		
		System.out.println("5) ----------------");
		NDManager m2 = NDManager.newBaseManager();
		NDArray y = NDArray.decode(m2, xb);
		System.out.println(y);
		m2.close();
	}
}

