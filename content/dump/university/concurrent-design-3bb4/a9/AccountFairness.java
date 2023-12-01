
// As the problem is stated above, there is no fairness among processes
// withdrawing an amount.
// In particular, a process withdrawing a large amount that may only be
// satisfied by several deposits will starve
// if smaller withdrawals are served immediately. Now, modify the implementation
// such that withdrawals follow the first-come-first-serve discipline. State the
// monitor invariant!
//
// Testing is now modified to ensure that 300 is attempted to be withdrawn
// before 100. A valid log is D100D180D120W300W100.
//
// Hint: Use the idea of the ticket algorithm. Remember that each process
// calling withdraw and deposit has its stack and, therefore, its copy of local
// variables.
// The solution requires about ten lines of code.
import java.util.regex.Pattern;

class Account {
  // invariants: balance >= 0 && next <= ticket
  private int balance = 0; // balance >= 0
  private int number = 0;
  private int next = 0; // number >= next >= 0

  synchronized void deposit(int amount) {
    balance += amount;
    TestAccount.log += "D" + amount;
    notifyAll();
  }

  synchronized void withdraw(int amount) {
    int ticket = number++;
    while (ticket != next || balance < amount) {
      try {
        wait(); // wait for turn or sufficient balance
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
    balance -= amount;
    TestAccount.log += "W" + amount;
    next++;
    notifyAll();
  }
}

class Depositer extends Thread {
  Account a;
  int amount;

  Depositer(Account a, int amount) {
    this.a = a;
    this.amount = amount;
  }

  public void run() {
    a.deposit(amount);
  }
}

class Withdrawer extends Thread {
  Account a;
  int amount;

  Withdrawer(Account a, int amount) {
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
    Account a = new Account();
    Withdrawer w0 = new Withdrawer(a, 300);
    w0.start();
    // wait 10 ms to ensure that w0 tries to withdraw before w1
    try {
      Thread.sleep(10);
    } catch (Exception x) {
    }
    Withdrawer w1 = new Withdrawer(a, 100);
    w1.start();
    Depositer d0 = new Depositer(a, 100);
    d0.start();
    try {
      Thread.sleep(10);
    } catch (Exception x) {
    }
    Depositer d1 = new Depositer(a, 120);
    d1.start();
    Depositer d2 = new Depositer(a, 180);
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
    assert Pattern.matches(".*W300.*W100.*", log) : "300 must be withdrawn before 100 withdrawn";
    assert Pattern.matches(".*D120.*W300.*", log) : "120 must be deposited before 300 withdrawn";
    assert Pattern.matches(".*D180.*W300.*", log) : "180 must be deposited before 300 withdrawn";
    assert Pattern.matches(".*D100.*W100.*", log) : "120 must be deposited before 100 withdrawn";
  }
}
