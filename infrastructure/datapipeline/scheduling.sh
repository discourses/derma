#!/bin/sh

# In progress

# Delete all temporary crontab files
rm -f /tmp/crontab*

# Create a temporary crontab file
CRON_FILE="/tmp/crontab"
if [ ! -f $CRON_FILE ]; then
  echo "A root cron does not exist.  Creating ..."
  touch $CRON_FILE
  /usr/bin/crontab $CRON_FILE
fi

# Set-up the contents of the crontab file
grep -qi "bootstrap" $CRON_FILE
if [ $? != 0 ]; then
  echo "Appending bootstrap cron job"
  /bin/echo "SHELL=/bin/bash" >> $CRON_FILE
  /bin/echo "PATH=/bin:/sbin:/usr/bin:/usr/sbin:~:/tmp" >> $CRON_FILE
  /bin/echo "/1 * * * * ~/....sh" >> $CRON_FILE
fi
