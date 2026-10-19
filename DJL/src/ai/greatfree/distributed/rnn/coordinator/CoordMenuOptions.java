package ai.greatfree.distributed.rnn.coordinator;

/**
 * Created: 09/01/2024, Bing Li
 */
final class CoordMenuOptions
{
	public final static int NO_OPTION = -1;
	public final static int QUIT = 0;
	public final static int DISTRIBUTE_TRAINING_DATA = 1;
	public final static int INITIALIZAE = 2;
	public final static int START_TRAINING = 3;
	public final static int PREDICT = 4;
	public final static int SHUTDOWN_CHILDREN = 5;
	public final static int SHUTDOWN_ROOT = 6;
	public final static int SHUTDOWN_REGISTRY_SERVER = 7;
}

