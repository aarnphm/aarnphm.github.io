# Google Workspace SSO

<Callout variant="gated-feature">
Google Workspace SSO is available on the <a href="/pricing">Enterprise plan</a>. Contact <a href="mailto:sales@modal.com">sales@modal.com</a> for more information.
</Callout>

This guide configures Google Workspace as a SAML identity provider for your
Modal Workspace. It's separate from signing in to Modal with "Continue with
Google".

## Prerequisites

- A Workspace that's on an [Enterprise](/pricing) plan
- Admin access to the Workspace you want to configure with Google Workspace
  Single-Sign-On (SSO)
- Super administrator privileges for your Google Workspace organization

## Supported features

- Identity Provider (IdP) initiated SSO
- Service Provider (SP) initiated SSO
- Just-In-Time account provisioning

## Configuration

### Read this before you enable "Require SSO"

Enabling "Require SSO" will force all users to sign in via Google Workspace.
Ensure that you have admin access to your Modal Workspace through a Google
Workspace account before enabling.

### Configuration steps

#### Step 1: Add Modal app to Google Workspace

1. Sign in to the [Google Admin console](https://admin.google.com/) as a super
   administrator.
2. Navigate to "Apps > Web and mobile apps".
3. Select "Add app > Add custom SAML app".
4. Name the application `Modal` and click "Continue".
5. On the "Google Identity Provider details" page, click "Download Metadata"
   to save the IdP metadata XML file, then click "Continue". You'll upload this
   file to Modal in Step 4.

For more information, see Google's guide to
[setting up a custom SAML app](https://support.google.com/a/answer/6087519).

#### Step 2: Configure the Modal SAML settings

On the "Service provider details" page, enter the following values. Replace
`<workspace>` with your Modal Workspace name.

| Setting         | Value                                             |
| --------------- | ------------------------------------------------- |
| ACS URL         | `https://modal.com/api/okta/saml/sso/<workspace>` |
| Entity ID       | `https://www.modal.com`                           |
| Start URL       | Leave empty                                       |
| Signed response | Leave unchecked                                   |
| Name ID format  | `EMAIL`                                           |
| Name ID         | Basic Information > Primary email                 |

<Callout variant="warning">
Sign-in fails if "Start URL" has a value or "Signed response" is checked.
</Callout>

The Entity ID must match exactly, including `https://` and `www`, with no
trailing slash. If you later
[rename your Workspace](/docs/guide/workspaces#renaming-a-workspace), update
the ACS URL to the new name.

Click "Continue".

#### Step 3: Configure SAML attribute mapping

On the "Attribute mapping" page, click "Add mapping" for each row below.
Select the value as the Google Directory attribute, and enter the name as the
app attribute exactly as shown.

| Name      | Value                             |
| --------- | --------------------------------- |
| email     | Basic Information > Primary email |
| firstName | Basic Information > First name    |
| lastName  | Basic Information > Last name     |

Click "Finish" to create the application.

#### Step 4: Link your Workspace to Google Workspace Modal application

Google Workspace doesn't publish a metadata URL, so you upload the metadata
file from Step 1 instead.

1. Sign in to https://modal.com and visit your [Workspace Management](/settings/workspace-management/identity-and-provisioning) page's "Identity and Provisioning" tab.
2. Click "Upload XML file" and select the metadata file you downloaded in
   Step 1.

If you didn't download the file in Step 1,
open the Modal application in the Google Admin console and click "Download
Metadata" on its details page.

#### Step 5: Assign users / groups and test the integration

1. Navigate back to your Modal application in the Google Admin console.
2. Open "User access" and turn the app on for everyone, or for the
   appropriate organizational units or groups. See Google's guide to
   [turning on a SAML app](https://support.google.com/a/answer/6087519#zippy=%2Cstep-turn-on-your-saml-app)
   for detailed steps.
3. To test the integration, sign in to Google as one of the users you turned
   the app on for.
4. Open the Modal application from the Google apps menu to initiate Single
   Sign-On. Administrators can also click "Test SAML login" on the
   application's page in the Google Admin console.
5. Confirm that the user is added to the expected Modal Workspace before
   enabling "Require SSO".

## SP-initiated SSO

The sign-in process is initiated from https://modal.com/login/sso

1. Enter your Workspace name in the input
2. Click "Continue with SSO" to authenticate with Google Workspace

## Troubleshooting

### Sign-in fails after Google authenticates the user

Confirm that the attribute mapping contains `email`, `firstName`, and
`lastName` with the exact capitalization shown above, and that the affected
user has a first name, last name, and primary email in Google Workspace.

### Sign-in stops working after the Google certificate is renewed

Modal reads the metadata file you uploaded, so it doesn't pick up changes made
in Google Workspace. After rotating the application's SAML certificate in the
Google Admin console, download the metadata file again and upload it as
described in Step 4.
