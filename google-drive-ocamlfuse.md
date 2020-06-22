FUSE filesystem over Google Drive
=================================

[![Join the chat at https://gitter.im/google-drive-ocamlfuse/Lobby](https://badges.gitter.im/google-drive-ocamlfuse/Lobby.svg)](https://gitter.im/google-drive-ocamlfuse/Lobby)
[![Docker Pulls](https://img.shields.io/docker/pulls/maltokyo/docker-google-drive-ocamlfuse)](https://hub.docker.com/r/maltokyo/docker-google-drive-ocamlfuse)

**google-drive-ocamlfuse** is a FUSE filesystem for Google Drive,
written in OCaml. It lets you mount your Google Drive on Linux.

### Features (see [what's new](https://github.com/astrada/google-drive-ocamlfuse/wiki/What%27s-new))

* Full read/write access to ordinary files and folders
* Read-only access to Google Docs, Sheets, and Slides (exported to
  configurable formats)
* Multiple account support
* Duplicate file handling
* Access to trash (`.Trash` directory)
* Unix permissions and ownership
* Symbolic links
* Read-ahead buffers when streaming
* Accessing content shared with you (requires [configuration](doc/Configuration.md))
* Team Drive [Support](https://github.com/astrada/google-drive-ocamlfuse/wiki/Team-Drives)
* Service Account [Support](https://github.com/astrada/google-drive-ocamlfuse/wiki/Service-Accounts)
* OAuth2 for Devices [Support](https://github.com/astrada/google-drive-ocamlfuse/wiki/OAuth2-for-Devices)

### Resources

* [Homepage](https://astrada.github.io/google-drive-ocamlfuse/)
* [Wiki](https://github.com/astrada/google-drive-ocamlfuse/wiki): includes
  installation instructions, and more details about configuration, and
  authorization

### Authorization

Please be sure to have a look at the
[authorization](https://github.com/astrada/google-drive-ocamlfuse/wiki/Authorization)
page, to understand how the authorization process works, and to discover all
the available options.

Getting started
---------------

### Installation

I've uploaded .deb packages for Ubuntu to my
[PPA](https://launchpad.net/~alessandro-strada/+archive/ppa). In order to to
install it, use the commands below:

    sudo add-apt-repository ppa:alessandro-strada/ppa
    sudo apt-get update
    sudo apt-get install google-drive-ocamlfuse

New beta versions are available on this
[PPA](https://launchpad.net/~alessandro-strada/+archive/ubuntu/google-drive-ocamlfuse-beta).
If you want to test them, use the commands below:

    sudo add-apt-repository ppa:alessandro-strada/google-drive-ocamlfuse-beta
    sudo apt-get update
    sudo apt-get install google-drive-ocamlfuse

For other installation options, please refer to the [wiki](https://github.com/astrada/google-drive-ocamlfuse/wiki/Installation).

How to build
------------

### Requirements

* [OCaml][] >= 4.02.3
* [Findlib][] >= 1.2.7
* [ocamlfuse][] >= 2.7.1
* [gapi-ocaml][] >= 0.3.6
* [sqlite3-ocaml][] >= 1.6.1

[OCaml]: http://caml.inria.fr/ocaml/release.en.html
[Findlib]: http://projects.camlcity.org/projects/findlib.html/
[ocamlfuse]: https://github.com/astrada/ocamlfuse
[gapi-ocaml]: https://github.com/astrada/gapi-ocaml
[sqlite3-ocaml]: https://mmottl.github.io/sqlite3-ocaml/

### Configuration and installation

To build the executable, run

    dune build @install

To install it, run (as root, if your user doesn't have enough privileges)

    dune install

To uninstall anything that was previously installed, execute

    dune uninstall

Usage
-----

The first time, you can run `google-drive-ocamlfuse` without parameters:

    google-drive-ocamlfuse

This command will create the default application directory
(`~/.gdfuse/default`), containing the configuration file `config` (see the
[wiki
page](https://github.com/astrada/google-drive-ocamlfuse/wiki/Configuration)
for more details about configuration). And it will start a web browser to
obtain authorization to access your Google Drive. This will let you modify
default configuration before mounting the filesystem.

Then you can choose a local directory to mount your Google Drive (e.g.: `~/GoogleDrive`).

Create the mount point, if it doesn't exists:

    mkdir ~/GoogleDrive

Then you can mount the filesystem (replacing [mountpoint] with the name of your desired folder):

    google-drive-ocamlfuse [mountpoint]

If you have more than one account, you can run:

    google-drive-ocamlfuse -label label [mountpoint]

Using `label` to distinguish different accounts. The program will use the
directory `~/.gdfuse/label` to host configuration, application state, and file
cache. No file is shared among different accounts, so you can have a different
configuration for each one.

To unmount the filesystem, issue this command:

    fusermount -u mountpoint

### Troubleshooting

This application is still under testing, so there are probably bugs to
discover and fix. To be extra sure, if you want, you can mount the filesystem
in read-only mode, modifying the configuration (see the
[documentation](https://github.com/astrada/google-drive-ocamlfuse/wiki/Configuration)),
to avoid any write attempt to the server. Anyway, the `rm` command will simply
trash your file, so you should always be able to rollback any changes. If you
have problems, you can turn on debug logging:

    google-drive-ocamlfuse -debug mountpoint

In `~/.gdfuse/default` you can find `curl.log` that will track every request
to the Google Drive API, and `gdfuse.log` that will log FUSE operations and
cache management. If something goes wrong, you can try clearing the cache,
with this command:

    google-drive-ocamlfuse -cc

If something still doesn't work, try starting from scratch removing everything
in `~/.gdfuse/default`. In this case you will need to reauthorize the
application.

Note that in order to reduce latency, the application will query the server
and check for changes only every 60 seconds (configurable). So, if you make a
change to your documents (server side), you won't see it immediately in the
mounted filesystem.

Note also that Google Documents will be exported read-only.

### Support

If you have questions, suggestions or want to report a problem, you may want
to open an [issue](https://github.com/astrada/google-drive-ocamlfuse/issues)
on github.
