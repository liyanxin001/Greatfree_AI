package ai.greatfree.computation;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;

/**
 * Created: 08/19/2024, Bing Li
 */
final class CustomInitialization
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.randomUniform(0, 1, new Shape(2, 4));

			SequentialBlock net = BuiltInInitialization.getNet();
			net.setInitializer(new MyInit(), Parameter.Type.WEIGHT);
			net.initialize(manager, DataType.FLOAT32, x.getShape());
			Block linearLayer = net.getChildren().valueAt(0);
			NDArray weight = linearLayer.getParameters().valueAt(0).getArray();
			System.out.println(weight);

			System.out.println("------------------------");
			// '__'i() is an inplace operation to modify the original NDArray
			NDArray weightLayer = net.getChildren().valueAt(0).getParameters().valueAt(0).getArray();
			weightLayer.addi(7);
			weightLayer.divi(9);
			weightLayer.set(new NDIndex(0, 0), 2020); // set the (0, 0) index to 2020
			System.out.println(weightLayer);
		}
	}
}
