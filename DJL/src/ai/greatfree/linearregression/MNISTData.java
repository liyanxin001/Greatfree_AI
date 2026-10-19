package ai.greatfree.linearregression;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.logging.Logger;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Record;
import ai.greatfree.utils.ImageUtils;

/**
 * Created: 08/14/2024, Bing Li
 */
final class MNISTData
{
	private final static Logger log = Logger.getLogger("ai.greatfree.linearregression");

	// Saved in the FashionMnist class for later use
	public static String[] getFashionMnistLabels(int[] labelIndices)
	{
		String[] textLabels = { "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot" };
		String[] convertedLabels = new String[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++)
		{
			convertedLabels[i] = textLabels[labelIndices[i]];
		}
		return convertedLabels;
	}

	public static String getFashionMnistLabel(int labelIndice)
	{
		String[] textLabels = { "t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot" };
		return textLabels[labelIndice];
	}

	// Saved in the FashionMnistUtils class for later use
	public static BufferedImage showImages(ArrayDataset dataset, int number, int width, int height, int scale, NDManager manager)
	{
		// Plot a list of images
		BufferedImage[] images = new BufferedImage[number];
		String[] labels = new String[number];
		for (int i = 0; i < number; i++)
		{
			Record record = dataset.get(manager, i);
			NDArray array = record.getData().get(0).squeeze(-1);
			int y = (int) record.getLabels().get(0).getFloat();
			images[i] = toImage(array, width, height);
			labels[i] = getFashionMnistLabel(y);
		}
		int w = images[0].getWidth() * scale;
		int h = images[0].getHeight() * scale;

		log.info("w = " + w);
		log.info("h = " + h);
		log.info("images length = " + images.length);
		
		for (int i = 0; i < labels.length; i++)
		{
			log.info(i + ") " + labels[i]);
		}

		return ImageUtils.showImages(images, labels, w, h);
	}

	private static BufferedImage toImage(NDArray array, int width, int height)
	{
		System.setProperty("apple.awt.UIElement", "true");
		BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = (Graphics2D) img.getGraphics();
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				float c = array.getFloat(j, i) / 255; // scale down to between 0 and 1
				g.setColor(new Color(c, c, c)); // set as a gray color
				g.fillRect(i, j, 1, 1);
			}
		}
		g.dispose();
		return img;
	}
}
