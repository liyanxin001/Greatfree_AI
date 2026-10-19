package ai.greatfree.preliminaries;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.components.Marker;
import tech.tablesaw.plotly.traces.ScatterTrace;

/**
 * Created: 08/13/2024, Bing Li
 */
final class VisualProbability
{
	public static void main(String[] args)
	{
		try (NDManager manager = NDManager.newBaseManager())
		{
			float[] fairProbsArr = new float[6];
			for (int i = 0; i < fairProbsArr.length; i++)
			{
				fairProbsArr[i] = 1f / 6;
			}
			NDArray fairProbs = manager.create(fairProbsArr);

			NDArray counts = manager.randomMultinomial(10, fairProbs, new Shape(500));
			NDArray cumCounts = counts.cumSum(0);
			NDArray estimates = cumCounts.div(cumCounts.sum(new int[] { 1 }, true));

			int height = 500;
			int width = 700;
			String xLabel = "Group of experiments";
			String yLabel = "Estimated probability";

			Layout layout = Layout.builder().height(height).width(width).showLegend(true).xAxis(Axis.builder().title(xLabel).build()).yAxis(Axis.builder().title(yLabel).build()).build();

			ScatterTrace[] traces = new ScatterTrace[7];
			double[] x;
			double[] y;
			for (int i = 0; i < 6; i++)
			{
				// x = Functions.floatToDoubleArray(manager.arange(500f).toFloatArray());
				// y = Functions.floatToDoubleArray(estimates.get(new NDIndex(":,
				// "+i)).toFloatArray());
				x = convertFloatsToDoubles(manager.arange(500f).toFloatArray());
				y = convertFloatsToDoubles(estimates.get(new NDIndex(":, " + i)).toFloatArray());
				traces[i] = ScatterTrace.builder(x, y).mode(ScatterTrace.Mode.LINE).name("P(die=" + (i + 1) + ")").build();
			}

			x = new double[500];
			y = new double[500];
			for (int i = 0; i < 500; i++)
			{
				x[i] = i;
				y[i] = 0.167;
			}
			traces[6] = ScatterTrace.builder(x, y).mode(ScatterTrace.Mode.LINE).name("Underlying Probability").marker(Marker.builder().color("black").build()).build();
			Plot.show(new Figure(layout, traces));
		}
	}

	public static double[] convertFloatsToDoubles(float[] input)
	{
		if (input == null)
		{
			return null; // Or throw an exception - your choice
		}
		double[] output = new double[input.length];
		for (int i = 0; i < input.length; i++)
		{
			output[i] = input[i];
		}
		return output;
	}
}
