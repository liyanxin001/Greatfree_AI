package ai.greatfree.performance;

import java.util.ArrayList;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.greatfree.utils.StopWatch;

/**
 * Created: 08/22/2024, Bing Li
 */
final class ParallelComm
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x_cpu = manager.randomUniform(0f, 1f, new Shape(2000, 2000), DataType.FLOAT32, Device.cpu());
			NDArray x_gpu = manager.randomUniform(0f, 1f, new Shape(6000, 6000), DataType.FLOAT32, Device.gpu());

			// calculating GPU computation time
			StopWatch stopWatch = new StopWatch();
			stopWatch.start();

			System.out.println(x_cpu);
			NDArray Y = PerUtils.run(x_gpu);

			stopWatch.stop();
			ArrayList<Double> times = stopWatch.getTimes();
			System.out.println("Run on GPU: " + times.get(times.size() - 1) + " nanoseconds ");

			// calculating copy to CPU time
			StopWatch stopWatch1 = new StopWatch();
			stopWatch1.start();

			NDArray y_cpu = PerUtils.copyToCPU(Y);
			System.out.println(y_cpu);

			stopWatch1.stop();
			times = stopWatch1.getTimes();
			System.out.println("Copy to CPU: " + times.get(times.size() - 1) + " nanoseconds ");
		}
	}
}
