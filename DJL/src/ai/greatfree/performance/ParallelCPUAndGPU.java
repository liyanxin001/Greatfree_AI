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
final class ParallelCPUAndGPU
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x_cpu = manager.randomUniform(0f, 1f, new Shape(2000, 2000), DataType.FLOAT32, Device.cpu());
			NDArray x_gpu = manager.randomUniform(0f, 1f, new Shape(6000, 6000), DataType.FLOAT32, Device.gpu());

			// initial warm up of devices
			PerUtils.run(x_cpu);
			PerUtils.run(x_gpu);

			// calculating CPU computation time
			StopWatch stopWatch0 = new StopWatch();
			stopWatch0.start();

			PerUtils.run(x_cpu);

			stopWatch0.stop();
			ArrayList<Double> times = stopWatch0.getTimes();
			System.out.println("CPU time: " + times.get(times.size() - 1) + " nanoseconds ");

			// calculating GPU computation time
			StopWatch stopWatch1 = new StopWatch();
			stopWatch1.start();

			PerUtils.run(x_gpu);

			stopWatch1.stop();
			times = stopWatch1.getTimes();
			System.out.println("GPU time: " + times.get(times.size() - 1) + " nanoseconds ");
		}
	}
}


