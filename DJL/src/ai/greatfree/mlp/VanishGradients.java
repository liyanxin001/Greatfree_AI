package ai.greatfree.mlp;

import java.util.Arrays;

import org.apache.commons.lang3.ArrayUtils;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/18/2024, Bing Li
 */
final class VanishGradients
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(-8.0f, 8.0f, 0.1f);
			x.setRequiresGradient(true);
			NDArray y = null;
			try (GradientCollector gc = Engine.getInstance().newGradientCollector())
			{
				y = Activation.sigmoid(x);
				gc.backward(y);
			}

			NDArray res = x.getGradient();

			int xLength = (int) x.size();
			int yLength = (int) y.size();

			float[] X = new float[xLength];
			float[] Y = new float[yLength];
			float[] Z = new float[yLength];

			X = x.toFloatArray();
			Y = y.toFloatArray();
			Z = res.toFloatArray();

			String[] groups = new String[xLength * 2];
			Arrays.fill(groups, 0, xLength, "sigmoid");
			Arrays.fill(groups, xLength, xLength * 2, "gradient");

			Table data = Table.create("Data").addColumns(FloatColumn.create("X", ArrayUtils.addAll(X, X)), FloatColumn.create("grad of relu", ArrayUtils.addAll(Y, Z)), StringColumn.create("groups", groups));
//			render(LinePlot.create("", data, "x", "grad of relu", "groups"), "text/html");
			Plot.show(LinePlot.create("", data, "x", "grad of relu", "groups"));
		}
	}
}

