package ai.greatfree.performance;

import ai.djl.Device;
import ai.djl.engine.Engine;

/**
 * 
 * @author libing
 * 
 *         12/04/2024
 *
 */
final class ComputingDevices
{
	public static void main(String[] args)
	{
		System.out.println(Device.cpu());
		System.out.println(Device.gpu());
		System.out.println(Device.gpu(1));
		
		System.out.println("GPU count: " + Engine.getInstance().getGpuCount());
		Device d = Device.gpu(1);
		System.out.println(d);
	}
}

