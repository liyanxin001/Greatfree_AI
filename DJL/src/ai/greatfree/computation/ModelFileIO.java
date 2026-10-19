package ai.greatfree.computation;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Created: 08/19/2024, Bing Li
 */
final class ModelFileIO
{
	public static SequentialBlock createMLP()
	{
		SequentialBlock mlp = new SequentialBlock();
		mlp.add(Linear.builder().setUnits(256).build());
		mlp.add(Activation.reluBlock());
		mlp.add(Linear.builder().setUnits(10).build());
		return mlp;
	}

	public static void main(String[] args) throws IOException, MalformedModelException
	{
		String paramFile = "/Users/libing/Temp/param-file";
		SequentialBlock original = createMLP();
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.randomUniform(0, 1, new Shape(2, 5));
			original.initialize(manager, DataType.FLOAT32, x.getShape());
			ParameterStore ps = new ParameterStore(manager, false);
			NDArray y = original.forward(ps, new NDList(x), false).singletonOrThrow();
			System.out.println(y);

			// Save file
			File mlpParamFile = new File(paramFile);
			DataOutputStream os = new DataOutputStream(Files.newOutputStream(mlpParamFile.toPath()));
			original.saveParameters(os);

			// Create duplicate of network architecture
			SequentialBlock clone = createMLP();
			// Load Parameters
			clone.loadParameters(manager, new DataInputStream(Files.newInputStream(mlpParamFile.toPath())));

			// Original model's parameters
			PairList<String, Parameter> originalParams = original.getParameters();
			// Loaded model's parameters
			PairList<String, Parameter> loadedParams = clone.getParameters();

			for (int i = 0; i < originalParams.size(); i++)
			{
				if (originalParams.valueAt(i).getArray().equals(loadedParams.valueAt(i).getArray()))
				{
					System.out.printf("True ");
				}
				else
				{
					System.out.printf("False ");
				}
			}
		}
	}
}

