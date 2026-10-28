package ai.greatfree.distributed.cnn.registry;

/*    */ import java.io.IOException;
/*    */ import org.greatfree.exceptions.RemoteReadException;
/*    */ import org.greatfree.server.container.ServerContainer;
/*    */ import org.greatfree.server.container.ServerTask;
/*    */ import org.greatfree.util.TerminateSignal;
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */ public class RegisterServer
/*    */ {
/*    */   private ServerContainer server;
/* 19 */   private static RegisterServer instance = new RegisterServer();
/*    */ 
/*    */   
/*    */   public static RegisterServer CS() {
/* 23 */     if (instance == null) {
/*    */       
/* 25 */       instance = new RegisterServer();
/* 26 */       return instance;
/*    */     } 
/*    */ 
/*    */     
/* 30 */     return instance;
/*    */   }
/*    */ 
/*    */ 
/*    */ 
/*    */   
/*    */   public void stop(long timeout) throws ClassNotFoundException, IOException, InterruptedException, RemoteReadException {
/* 37 */     TerminateSignal.SIGNAL().notifyAllTermination();
/* 38 */     Register.unregister();
/* 39 */     this.server.stop(timeout);
/*    */   }
/*    */ 
/*    */   
/*    */   public void start(int port, ServerTask task) throws IOException, ClassNotFoundException, RemoteReadException {
/* 44 */     Register.register();
/* 45 */     this.server = new ServerContainer(port, task);
/* 46 */     this.server.start();
/*    */   }
/*    */ 
/*    */ 
/*    */ 
/*    */ 
/*    */   
/*    */   public void start(int port) throws IOException, ClassNotFoundException, RemoteReadException {
/* 54 */     Register.register();
/* 55 */     this.server = new ServerContainer(port, new RegistryTask());
/* 56 */     this.server.start();
/*    */   }
/*    */ 
/*    */   
/*    */   public void start(ServerTask task, String configXML) throws IOException, ClassNotFoundException, RemoteReadException {
/* 61 */     Register.register();
/* 62 */     this.server = new ServerContainer(task, configXML);
/* 63 */     this.server.start();
/*    */   }
/*    */ }
