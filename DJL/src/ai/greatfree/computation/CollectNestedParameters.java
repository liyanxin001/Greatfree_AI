package ai.greatfree.computation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;

/**
 * Created: 08/19/2024, Bing Li
 */
final class CollectNestedParameters
{
	public static SequentialBlock block1()
	{
		SequentialBlock net = new SequentialBlock();
		net.add(Linear.builder().setUnits(32).build());
		net.add(Activation.reluBlock());
		net.add(Linear.builder().setUnits(16).build());
		net.add(Activation.reluBlock());
		return net;
	}

	public static SequentialBlock block2()
	{
		SequentialBlock net = new SequentialBlock();
		for (int i = 0; i < 4; i++)
		{
			net.add(block1());
		}
		return net;
	}

	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			int inputSize = 20;
			NDArray x = manager.randomUniform(0, 1, new Shape(2, inputSize)); // (2, 20) shape

			SequentialBlock rgnet = new SequentialBlock();
			rgnet.add(block2());
			rgnet.add(Linear.builder().setUnits(10).build());
			rgnet.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			rgnet.initialize(manager, DataType.FLOAT32, x.getShape());

			ParameterStore ps = new ParameterStore(manager, false);
			System.out.println(rgnet.forward(ps, new NDList(x), false).singletonOrThrow());

			for (var param : rgnet.getParameters())
			{
				System.out.println(param.getValue().getArray());
			}
			
			Block majorBlock1 = rgnet.getChildren().get(0).getValue();
			Block subBlock2 = majorBlock1.getChildren().valueAt(1);
			Block linearLayer1 = subBlock2.getChildren().valueAt(0);
			NDArray bias = linearLayer1.getParameters().valueAt(1).getArray();
			System.out.println(bias);
		}
	}
}

