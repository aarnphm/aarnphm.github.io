
// A bank account is shared by several people (processes).
// Each person may deposit or withdraw funds from the account.
// The current balance in the account is the sum of all deposits to date minus the sum of all withdrawals.
// The balance must never become negative.
// A process that withdraws an amount that would make the balance negative has to wait until the balance is large enough.
//
// Implement a monitor in Java to solve the problem. The monitor has two
// procedures (methods): deposit(amount) and withdraw(amount), for positive integer amount.
// If the balance is 0 and processes are requesting to withdraw 300 and 100 and depositing 100, 120, and 180,
// in that order, an acceptable solution is that the process requesting 100 will be served right after another process
// deposits 100, even if the request for 300 came before the request for 100. That is, requests should be served as
// soon as a request can be served. Do not use Java library classes for Account (there is no need; the solution is short),
// though you may use whatever you like for testing. State the monitor invariant!
//
// The implementation below is instrumented with statements that log the
// execution of deposit() and withdraw(). This is used for testing that all deposits and withdrawals occur and occur in
// a valid order. For example, the log could be D100D180D120W100W300.
//
import java.util.regex.Pattern;

class Account {
  // needs a balance
  private int balance = 0;

  // invariants: balance >= 0

  synchronized void deposit(int amount) {
    balance += amount;
    TestAccount.log += "D" + amount;
    notifyAll();
  }

  synchronized void withdraw(int amount) {
    while (balance < amount) {
      try {
        wait(); // if balance is insufficient, then blocking thread to wait for deposit
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    balance -= amount;
    TestAccount.log += "W" + amount;
  }
}

class Depositer extends Thread {
  AccountFairness a;
  int amount;

  Depositer(AccountFairness a, int amount) {
    this.a = a;
    this.amount = amount;
  }

  public void run() {
    a.deposit(amount);
  }
}

class Withdrawer extends Thread {
  AccountFairness a;
  int amount;

  Withdrawer(AccountFairness a, int amount) {
    this.a = a;
    this.amount = amount;
  }

  public void run() {
    a.withdraw(amount);
  }
}

class TestAccount {
  static String log = "";

  public static void main(String[] args) {
    AccountFairness a = new AccountFairness();
    WithdrawerFairness w0 = new WithdrawerFairness(a, 300);
    w0.start();
    try {
      Thread.sleep(10);
    } catch (Exception x) {
    }
    WithdrawerFairness w1 = new WithdrawerFairness(a, 100);
    w1.start();
    DepositerFairness d0 = new DepositerFairness(a, 100);
    d0.start();
    try {
      Thread.sleep(10);
    } catch (Exception x) {
    }
    DepositerFairness d1 = new DepositerFairness(a, 120);
    d1.start();
    DepositerFairness d2 = new DepositerFairness(a, 180);
    d2.start();
    try {
      w0.join();
      w1.join();
      d0.join();
      d1.join();
      d2.join();
    } catch (Exception e) {
    }
    System.out.println(log);
    assert Pattern.matches(".*W300.*", log) : "300 not withdrawn";
    assert Pattern.matches(".*W100.*", log) : "100 not withdrawn";
    assert Pattern.matches(".*D100.*", log) : "100 not deposited";
    assert Pattern.matches(".*D120.*", log) : "120 not deposited";
    assert Pattern.matches(".*D180.*", log) : "180 not deposited";
    assert Pattern.matches(".*D....*W100.*", log) : "100 withdrawn before any deposit";
    assert Pattern.matches(".*(D120.*D180|D180.*D120).*W300.*", log)
        : "300 withdrawn before both 120 and 180 deposited";
  }
}
