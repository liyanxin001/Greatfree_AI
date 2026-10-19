package ai.greatfree.linearregression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.greatfree.utils.StopWatch;

/**
 * Created: 08/05/2024, Bing Li
 */
final class VectorSpeed
{
	public static void main(String[] args)
	{
		int n = 10000;
		NDManager manager = NDManager.newBaseManager();
		NDArray a = manager.ones(new Shape(n));
		NDArray b = manager.ones(new Shape(n));

		NDArray c = manager.zeros(new Shape(n));
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		for (int i = 0; i < n; i++)
		{
			c.set(new NDIndex(i), a.getFloat(i) + b.getFloat(i));
		}
//		NDArray d = a.add(b);
//		stopWatch.stop();
//		String.format("%.5f sec", stopWatch.stop());
//		System.out.println("Time Elapsed: " + stopWatch.getTime());
		System.out.println("Time Elapsed: " + stopWatch.stop());
		
		System.out.println("-------------------------");
//		stopWatch.reset();
		stopWatch.start();
		NDArray d = a.add(b);
		System.out.println(d);
//		stopWatch.stop();
//		System.out.println("Time Elapsed: " + stopWatch.getTime());
		System.out.println("Time Elapsed: " + stopWatch.stop());
	}
}

