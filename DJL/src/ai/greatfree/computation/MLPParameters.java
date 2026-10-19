package ai.greatfree.computation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.NormalInitializer;

/**
 * Created: 08/19/2024, Bing Li
 */
final class MLPParameters
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.randomUniform(0, 1, new Shape(2, 4));

			SequentialBlock net = new SequentialBlock();
			net.add(Linear.builder().setUnits(8).build());
			net.add(Activation.reluBlock());
			net.add(Linear.builder().setUnits(1).build());
			net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());

			ParameterStore ps = new ParameterStore(manager, false);
			System.out.println(net.forward(ps, new NDList(x), false).head()); // forward computation

			ParameterList params = net.getParameters();
			// Print out all the keys (unique!)
			for (var pair : params)
			{
				System.out.println(pair.getKey());
			}

			// Use the unique key to access the Parameter
			NDArray dense0Weight = params.get("01Linear_weight").getArray();
			NDArray dense0Bias = params.get("01Linear_bias").getArray();

			// Use indexing to access the Parameter
			NDArray dense1Weight = params.valueAt(2).getArray();
			NDArray dense1Bias = params.valueAt(3).getArray();

			System.out.println(dense0Weight);
			System.out.println(dense0Bias);

			System.out.println(dense1Weight);
			System.out.println(dense1Bias);
			
			System.out.println(dense0Weight.getGradient());
		}
	}
}
