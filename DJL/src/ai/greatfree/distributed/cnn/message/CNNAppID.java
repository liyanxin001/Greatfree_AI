package ai.greatfree.distributed.cnn.message;

public final class CNNAppID 
{
	public final static int DISTRIBUTE_TRAINING_DATA_REQUEST = 80000;
	public final static int DISTRIBUTE_TRAINING_DATA_RESPONSE = 80001;
	public final static int COLLECT_GRADIENTS_REQUEST = 80002;
	public final static int COLLECT_GRADIENTS_RESPONSE = 80003;
	public final static int INIT_REQUEST = 80004;
	public final static int INIT_RESPONSE = 80005;
	public final static int PREDICT_REQUEST = 80006;
	public final static int PREDICT_RESPONSE = 80007;
	public final static int START_TRAINING_NOTIFICATION = 80008;
    public final static int CHILD_PATHS_REQUEST = 80009;
    public final static int CHILD_PATHS_RESPONSE = 80010;
    public final static int TRAINING_DONE_NOTIFICATION = 80011;
    public final static int GRADIENTS_AGGREGATION_REQUEST = 80012;
	public final static int GRADIENTS_AGGREGATION_RESPONSE = 80013;
}
