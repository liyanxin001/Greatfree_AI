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
final class TiedParameters
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			SequentialBlock net = new SequentialBlock();

			// We need to give the shared layer a name
			// such that we can reference its parameters
			Block shared = Linear.builder().setUnits(8).build();
			SequentialBlock sharedRelu = new SequentialBlock();
			sharedRelu.add(shared);
			sharedRelu.add(Activation.reluBlock());

			net.add(Linear.builder().setUnits(8).build());
			net.add(Activation.reluBlock());
			net.add(sharedRelu);
			net.add(sharedRelu);
			net.add(Linear.builder().setUnits(10).build());

			NDArray x = manager.randomUniform(-10f, 10f, new Shape(2, 20), DataType.FLOAT32);

			net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());

			ParameterStore ps = new ParameterStore(manager, false);
			System.out.println(net.forward(ps, new NDList(x), false).singletonOrThrow());

			System.out.println("------------------------");
			// Check that the parameters are the same
			NDArray shared1 = net.getChildren().valueAt(2).getParameters().valueAt(0).getArray();
			NDArray shared2 = net.getChildren().valueAt(3).getParameters().valueAt(0).getArray();
			System.out.println(shared1.eq(shared2));
		}
	}
}
