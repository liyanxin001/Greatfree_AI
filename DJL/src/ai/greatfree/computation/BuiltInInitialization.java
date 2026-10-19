package ai.greatfree.computation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.initializer.XavierInitializer;

/**
 * Created: 08/19/2024, Bing Li
 */
final class BuiltInInitialization
{
	public static SequentialBlock getNet()
	{
		SequentialBlock net = new SequentialBlock();
		net.add(Linear.builder().setUnits(8).build());
		net.add(Activation.reluBlock());
		net.add(Linear.builder().setUnits(1).build());
		return net;
	}

	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.randomUniform(0, 1, new Shape(2, 4));
			SequentialBlock net = new SequentialBlock();
			net.add(Linear.builder().setUnits(8).build());

			net.setInitializer(new ConstantInitializer(1), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());

			Block linearLayer = net.getChildren().get(0).getValue();
			NDArray weight = linearLayer.getParameters().get(0).getValue().getArray();
			System.out.println(weight);

			System.out.println("------------------------");
			net = getNet();
			net.setInitializer(new ConstantInitializer(7777), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());
			linearLayer = net.getChildren().valueAt(0);
			weight = linearLayer.getParameters().valueAt(0).getArray();
			System.out.println(weight);

			System.out.println("------------------------");
			net = getNet();
			net.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());
			linearLayer = net.getChildren().valueAt(0);
			weight = linearLayer.getParameters().valueAt(0).getArray();
			System.out.println(weight);

			System.out.println("------------------------");
			net = new SequentialBlock();
			Linear linear1 = Linear.builder().setUnits(8).build();
			net.add(linear1);
			net.add(Activation.reluBlock());
			Linear linear2 = Linear.builder().setUnits(1).build();
			net.add(linear2);

			linear1.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
			linear1.initialize(manager, DataType.FLOAT32, x.getShape());

			linear2.setInitializer(Initializer.ZEROS, Parameter.Type.WEIGHT);
			linear2.initialize(manager, DataType.FLOAT32, x.getShape());

			System.out.println(linear1.getParameters().valueAt(0).getArray());
			System.out.println(linear2.getParameters().valueAt(0).getArray());

			System.out.println("------------------------");
			net = getNet();
			ParameterList params = net.getParameters();

			params.get("01Linear_weight").setInitializer(new NormalInitializer());
			params.get("03Linear_weight").setInitializer(Initializer.ONES);

			net.initialize(manager, DataType.FLOAT32, new Shape(2, 4));

			System.out.println(params.valueAt(0).getArray());
			System.out.println(params.valueAt(2).getArray());
		}
	}
}
