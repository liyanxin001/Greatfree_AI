package ai.greatfree.mlp;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Activation;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

/**
 * Created: 08/15/2024, Bing Li
 */
final class ReLUFunc
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			NDArray x = manager.arange(-8.0f, 8.0f, 0.1f);
			System.out.println(x);

			x.setRequiresGradient(true);

			System.out.println("--------------------");
			NDArray y = Activation.relu(x);
			System.out.println(y);

			// Converting the data into float arrays to render them in a plot.
			float[] X = x.toFloatArray();
			float[] Y = y.toFloatArray();

			Table data = Table.create("Data").addColumns(FloatColumn.create("X", X), FloatColumn.create("relu(x)", Y));
			// The figure is not shown up. 08/15/2024, Bing Li
//			Plot.show(LinePlot.create("", data, "x", "relu(X)"), "text/html");
			Plot.show(LinePlot.create("", data, "x", "relu(X)"));
		}
	}
}
